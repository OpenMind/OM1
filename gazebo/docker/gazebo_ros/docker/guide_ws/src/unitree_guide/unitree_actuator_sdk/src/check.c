#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>    //usleep()
#include <sys/time.h>
#include "LSerial.h"    //serial port communication
#include "motor_ctrl.h"

enum motor_command_type
{
    TORQUE,
    POSITION,
    VELOCITY
};

int main()
{
    int com_type = VELOCITY;
    float goal_T;
    float goal_W;
    float goal_Pos;
    float goal_K_W;
    float goal_K_P;

    if(com_type == TORQUE)
    {
        goal_T = 1;
        goal_W = 0;
        goal_Pos = 0;
        goal_K_W = 0;
        goal_K_P = 0;
    }
    else if(com_type == POSITION)
    {
        goal_T = 0;
        goal_W = 0;       //have to be 0
        goal_Pos = 0;
        goal_K_W = 3;       //K_W works as a dampping
        goal_K_P = 0.1;
    }
    else if(com_type == VELOCITY)
    {
        goal_T = 0;
        goal_W = 50.0;
        goal_Pos = 0;
        goal_K_W = 3;
        goal_K_P = 0;
    }
    else
    {
        printf("motor command type error!\n");
        return 1;
    }

    //Send
    MOTOR_send motor_s, motor_s1;
    // long long start_time = 0;
    motor_s.id = 0;                 //motor ID
    motor_s.mode = 10;              //switch to servo mode
    motor_s.T = goal_T;            //Nm, T<255.9
    motor_s.W = goal_W;               //rad/s, W<511.9
    motor_s.Pos = goal_Pos;          //rad, Pos<131071.9
    motor_s.K_P = goal_K_P;        //K_P<31.9     value should around 0.1
    motor_s.K_W = goal_K_W;        //K_W<63.9     value should around 3

    motor_s1.id = 0;
    motor_s1.mode = 0;
    //Receive
    MOTOR_recv motor_r;
    //over heat protection
    const float safe_torque = 0.7;
    const float cooldown_torque = 0.5;
    const long long overload_duration = 70000000;    //microsecond(us)
    const long long cooldown_duration = 600000000;    //microsecond(us)
    float safe_ratio = 1;
    // float past_torque = 0;
    // int overload_flag = 0;
    long long current = 0;
    long long overload_start = 0;
    long long cooldown_start = 0;
    enum motor_status
    {
        NORMAL,
        OVERHEAT,
        COOLDOWN
    };
printf("87\n");
#if defined(__linux__)
    int fd;
    fd = open_set("/dev/ttyUSB0");
#elif defined(__WIN32__)
    HANDLE fd;
    fd = open_set("\\\\.\\COM4");
#endif
printf("95\n");
    int status = NORMAL;

    modify_data(&motor_s);
    modify_data(&motor_s1);

    send_recv(fd, &motor_s, &motor_r);
    send_recv(fd, &motor_s1, &motor_r);
printf("102\n");
    extract_data(&motor_r);
    printf("START\n");

    for(int i=0; i<500; i++)
    {
        motor_s.T = goal_T;
        motor_s.K_P = goal_K_P;
        motor_s.K_W = goal_K_W;
        modify_data(&motor_s);

        if(status == NORMAL)
        {
            if(motor_r.T > safe_torque)
            {
                overload_start = getSystemTime();
                status = OVERHEAT;
            }
        }
        else if(status == OVERHEAT)
        {
            current = getSystemTime();
            if(motor_r.T < safe_torque)
            {
                status = NORMAL;
            }
            else if((current - overload_start) > overload_duration)
            {
                cooldown_start = getSystemTime();
                status = COOLDOWN;
            }
        }
        else if(status == COOLDOWN)
        {
            current = getSystemTime();
            if((current - cooldown_start) > cooldown_duration)
            {
                status = NORMAL;
            }
            else
            {
                safe_ratio = cooldown_torque / motor_r.T;
                motor_s.T = safe_ratio * goal_T;
                motor_s.K_P = safe_ratio * goal_K_P;
                motor_s.K_W = safe_ratio * goal_K_W;
                modify_data(&motor_s);
            }
        }
        printf("******************\n");
        printf("Torque command: %f\n", motor_s.T);
        send_recv(fd, &motor_s, &motor_r);
        extract_data(&motor_r);

        printf("status: %d\n", status);
        // printf("pos: %f\n", motor_r.Pos);
        printf("w: %f\n", motor_r.LW);
        printf("T: %f\n", motor_r.T);
        usleep(500000);
    }

    send_recv(fd, &motor_s1, &motor_r);
    extract_data(&motor_r);
    // show_resv_data_hex(&motor_r);
    printf("END\n");
    printf("ID: %d\n", motor_r.motor_id);
    // show_resv_data(&motor_r);

    close_serial(fd);
#if defined(__WIN32__)
    system("pause");
#endif

    return 0;
}
